// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

#include "detail/data_generation.hpp"
#include "detail/threading.hpp"

namespace roc::host_validation {
namespace {
uint64_t strideMagnitude(ptrdiff_t stride) {
    if (stride >= 0) return static_cast<uint64_t>(stride);
    return static_cast<uint64_t>(-(stride + 1)) + 1;
}

bool hasProvablyIndependentElements(const MutableTensorView& destination) {
    if (scalarTypeInfo(destination.type()).storageBits % 8 != 0) return false;

    std::vector<std::pair<uint64_t, size_t>> dimensions;
    dimensions.reserve(destination.shape().rank());
    for (size_t dimension = 0; dimension < destination.shape().rank(); ++dimension) {
        const size_t extent = destination.shape()[dimension];
        if (extent <= 1) continue;

        const uint64_t stride = strideMagnitude(destination.layout().strides()[dimension]);
        if (stride == 0) return false;
        dimensions.emplace_back(stride, extent);
    }
    std::ranges::sort(dimensions);

    uint64_t addressedSpan = 1;
    for (const auto& [stride, extent] : dimensions) {
        if (stride < addressedSpan) return false;
        const uint64_t additionalExtent = static_cast<uint64_t>(extent - 1);
        if (additionalExtent > (std::numeric_limits<uint64_t>::max() - addressedSpan) / stride)
            return false;
        addressedSpan += additionalExtent * stride;
    }
    return true;
}

#ifdef _OPENMP
void incrementLastDimensionFast(std::vector<size_t>& indices, const Shape& shape) {
    for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
        const size_t index = dimension - 1;
        if (++indices[index] < shape[index]) return;
        indices[index] = 0;
    }
}
#endif

void generateSerial(MutableTensorView destination, const GenerationOptions& options) {
    detail::forEachIndex(destination.shape(), [&](std::span<const size_t> indices, size_t) {
        const size_t logicalIndex =
            detail::logicalLinearIndex(indices, destination.shape(), options.indexOrder);
        detail::generateElement(destination, options, indices, logicalIndex);
    });
}

void generateParallel(MutableTensorView destination, const GenerationOptions& options,
                      int threadCount) {
#ifdef _OPENMP
    std::exception_ptr error;
    const size_t elementCount = destination.shape().elementCount();
#pragma omp parallel num_threads(threadCount)
    {
        try {
            const size_t threadIndex = static_cast<size_t>(omp_get_thread_num());
            const size_t actualThreadCount = static_cast<size_t>(omp_get_num_threads());
            const size_t baseCount = elementCount / actualThreadCount;
            const size_t remainder = elementCount % actualThreadCount;
            const size_t first = threadIndex * baseCount + std::min(threadIndex, remainder);
            const size_t count = baseCount + static_cast<size_t>(threadIndex < remainder);
            const size_t end = first + count;
            if (first != end) {
                std::vector<size_t> indices = detail::logicalCoordinates(
                    first, destination.shape(), LogicalIndexOrder::LastDimensionFastest);
                for (size_t traversalIndex = first; traversalIndex < end; ++traversalIndex) {
                    const size_t logicalIndex = detail::logicalLinearIndex(
                        indices, destination.shape(), options.indexOrder);
                    detail::generateElement(destination, options, indices, logicalIndex);
                    incrementLastDimensionFast(indices, destination.shape());
                }
            }
        } catch (...) {
#pragma omp critical(roc_host_validation_generation_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
#else
    (void)threadCount;
    generateSerial(destination, options);
#endif
}
}  // namespace

GenerationRunInfo generate(MutableTensorView destination, const GenerationOptions& options) {
    const size_t elementCount = destination.shape().elementCount();
    const int threadCount = hasProvablyIndependentElements(destination)
                                ? detail::operationThreadCount(elementCount)
                                : 1;
    if (threadCount == 1)
        generateSerial(destination, options);
    else
        generateParallel(destination, options, threadCount);
    return {.elementsGenerated = elementCount};
}

GenerationRunInfo generateAt(MutableTensorView destination, size_t logicalIndex,
                             const GenerationOptions& options) {
    const std::vector<size_t> indices =
        detail::logicalCoordinates(logicalIndex, destination.shape(), options.indexOrder);
    detail::generateElement(destination, options, indices, logicalIndex);
    return {.elementsGenerated = 1};
}
}  // namespace roc::host_validation
