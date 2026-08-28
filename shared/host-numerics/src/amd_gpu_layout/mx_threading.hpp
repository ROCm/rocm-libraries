// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace roc::host_numerics::amd_gpu_layout::detail {

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
void parallelForChunks(size_t iterationCount, size_t workItemCount, Function function) {
    if (iterationCount == 0) return;

    const int threadCount = operationThreadCount(workItemCount);
    if (threadCount <= 1) {
        function(0, iterationCount);
        return;
    }

#ifdef _OPENMP
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
#pragma omp parallel for schedule(static, 1) num_threads(threadCount)
    for (int chunkIndex = 0; chunkIndex < threadCount; ++chunkIndex) {
        if (failed.load(std::memory_order_relaxed)) continue;

        const size_t chunk = static_cast<size_t>(chunkIndex);
        const size_t chunks = static_cast<size_t>(threadCount);
        const size_t baseSize = iterationCount / chunks;
        const size_t extraItems = iterationCount % chunks;
        const size_t begin = chunk * baseSize + std::min(chunk, extraItems);
        const size_t end = begin + baseSize + static_cast<size_t>(chunk < extraItems);
        try {
            function(begin, end);
        } catch (...) {
#pragma omp critical(roc_host_numerics_amd_gpu_layout_exception)
            {
                if (!failure) failure = std::current_exception();
            }
            failed.store(true, std::memory_order_relaxed);
        }
    }
    if (failure) std::rethrow_exception(failure);
#else
    function(0, iterationCount);
#endif
}

}  // namespace roc::host_numerics::amd_gpu_layout::detail
