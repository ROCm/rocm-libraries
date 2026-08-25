// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <roc/host_validation/tensor.hpp>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace roc::host_validation::detail {
inline uint64_t strideMagnitude(ptrdiff_t stride) {
    if (stride >= 0) return static_cast<uint64_t>(stride);
    return static_cast<uint64_t>(-(stride + 1)) + 1;
}

inline bool hasProvablyIndependentElements(const Tensor& destination) {
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

inline bool storageOverlaps(const Tensor& left, const Tensor& right) {
    if (left.storage().empty() || right.storage().empty()) return false;
    const uintptr_t leftBegin = reinterpret_cast<uintptr_t>(left.storage().data());
    const uintptr_t rightBegin = reinterpret_cast<uintptr_t>(right.storage().data());
    const uintptr_t leftEnd = leftBegin + left.storage().size();
    const uintptr_t rightEnd = rightBegin + right.storage().size();
    return leftBegin < rightEnd && rightBegin < leftEnd;
}

inline size_t saturatedProduct(size_t left, size_t right) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        return std::numeric_limits<size_t>::max();
    return left * right;
}

inline int operationThreadCount(size_t workItemCount, size_t minimumWorkItemsPerThread = 4096,
                                int defaultMaximumThreadCount = 8) {
#ifdef _OPENMP
    if (workItemCount == 0 || omp_in_parallel()) return 1;

    const int runtimeMaximum = std::max(1, omp_get_max_threads());
    const char* configuredThreadCount = std::getenv("OMP_NUM_THREADS");
    const int maximum = configuredThreadCount != nullptr && configuredThreadCount[0] != '\0'
                            ? runtimeMaximum
                            : std::min(runtimeMaximum, defaultMaximumThreadCount);
    const size_t usefulThreadCount = std::max(
        size_t{1}, workItemCount / minimumWorkItemsPerThread +
                       static_cast<size_t>(workItemCount % minimumWorkItemsPerThread != 0));
    return static_cast<int>(std::min(usefulThreadCount, static_cast<size_t>(maximum)));
#else
    (void)workItemCount;
    (void)minimumWorkItemsPerThread;
    (void)defaultMaximumThreadCount;
    return 1;
#endif
}

template <typename Function>
void forEachParallelIndex(size_t count, size_t workItemCount, bool canParallelize,
                          size_t minimumWorkItemsPerThread, Function&& function) {
    const int threadCount = std::min<int>(
        canParallelize ? operationThreadCount(workItemCount, minimumWorkItemsPerThread) : 1,
        static_cast<int>(std::min(count, static_cast<size_t>(std::numeric_limits<int>::max()))));
#ifdef _OPENMP
    if (threadCount > 1 && count <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        std::exception_ptr error;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threadCount)
        for (ptrdiff_t index = 0; index < static_cast<ptrdiff_t>(count); ++index) {
            try {
                function(static_cast<size_t>(index));
            } catch (...) {
#pragma omp critical(roc_host_validation_parallel_error)
                {
                    if (!error) error = std::current_exception();
                }
            }
        }
        if (error) std::rethrow_exception(error);
        return;
    }
#else
    (void)threadCount;
#endif
    for (size_t index = 0; index < count; ++index) function(index);
}
}  // namespace roc::host_validation::detail
