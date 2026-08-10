// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#include "detail/data_generation.hpp"

namespace roc::host_validation {
GenerationRunInfo generate(MutableTensorView destination, const GenerationOptions& options) {
    detail::forEachIndex(destination.shape(), [&](std::span<const size_t> indices, size_t) {
        const size_t logicalIndex =
            detail::logicalLinearIndex(indices, destination.shape(), options.indexOrder);
        detail::generateElement(destination, options, indices, logicalIndex);
    });
    return {.elementsGenerated = destination.shape().elementCount()};
}

GenerationRunInfo generateAt(MutableTensorView destination, size_t logicalIndex,
                             const GenerationOptions& options) {
    const std::vector<size_t> indices =
        detail::logicalCoordinates(logicalIndex, destination.shape(), options.indexOrder);
    detail::generateElement(destination, options, indices, logicalIndex);
    return {.elementsGenerated = 1};
}
}  // namespace roc::host_validation
