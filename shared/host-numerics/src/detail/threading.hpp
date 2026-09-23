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
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor_storage.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace roc::host_numerics::detail {
struct IndexRange {
    size_t first;
    size_t pastLast;
};

inline IndexRange evenlyPartitionedRange(size_t count, size_t partitionCount, size_t partition) {
    if (partitionCount == 0 || partition >= partitionCount)
        throw std::invalid_argument("Parallel partition index is out of range.");
    const size_t elementsPerPartition = count / partitionCount;
    const size_t remainder = count % partitionCount;
    const size_t first = partition * elementsPerPartition + std::min(partition, remainder);
    return {
        first,
        first + elementsPerPartition + static_cast<size_t>(partition < remainder),
    };
}

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

template <typename Value, typename Transform, typename Reduction>
Value transformReduceParallelIndices(size_t count, size_t workItemCount, bool canParallelize,
                                     size_t minimumWorkItemsPerThread, Value initial,
                                     Transform&& transform, Reduction&& reduction) {
    if (count == 0) return initial;
    const int threadCount = std::min<int>(
        canParallelize ? operationThreadCount(workItemCount, minimumWorkItemsPerThread) : 1,
        static_cast<int>(std::min(count, static_cast<size_t>(std::numeric_limits<int>::max()))));
#ifdef _OPENMP
    if (threadCount > 1 && count <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        std::vector<Value> partials(static_cast<size_t>(threadCount), initial);
        std::exception_ptr error;
#pragma omp parallel num_threads(threadCount)
        {
            const size_t thread = static_cast<size_t>(omp_get_thread_num());
            Value partial = initial;
#pragma omp for schedule(static)
            for (ptrdiff_t index = 0; index < static_cast<ptrdiff_t>(count); ++index) {
                try {
                    partial = reduction(partial, transform(static_cast<size_t>(index)));
                } catch (...) {
#pragma omp critical(roc_host_numerics_parallel_error)
                    {
                        if (!error) error = std::current_exception();
                    }
                }
            }
            partials[thread] = partial;
        }
        if (error) std::rethrow_exception(error);
        for (const Value& partial : partials) initial = reduction(initial, partial);
        return initial;
    }
#else
    (void)threadCount;
#endif
    for (size_t index = 0; index < count; ++index) initial = reduction(initial, transform(index));
    return initial;
}
}  // namespace roc::host_numerics::detail
