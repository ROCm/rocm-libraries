// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <roc/host_numerics/tensor.hpp>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace roc::host_numerics::detail {
inline bool hasProvablyIndependentElements(const Tensor& destination) {
    if (scalarTypeInfo(destination.type()).storageBits % 8 != 0) return false;
    return detail::hasProvablyDistinctElementOffsets(destination.layout());
}

inline bool storageOverlaps(const Tensor& left, const Tensor& right) {
    return detail::byteRangesOverlap(left.rawEncodedBackingStorage(),
                                     right.rawEncodedBackingStorage());
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
#pragma omp parallel for schedule(static) num_threads(threadCount)
        for (ptrdiff_t index = 0; index < static_cast<ptrdiff_t>(count); ++index) {
            try {
                function(static_cast<size_t>(index));
            } catch (...) {
#pragma omp critical(roc_host_numerics_parallel_error)
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
}  // namespace roc::host_numerics::detail
