// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

namespace roc::hostnumerics::amd_gpu_layout::detail {

// Calls function(begin, end) on disjoint chunks covering [0, iterationCount).
// Chunks run on separate threads when `workItemCount` gives each thread at
// least 4096 items. Unless OMP_NUM_THREADS is set, at most 8 threads are used.
// `function` must be noexcept, because an exception cannot leave an OpenMP
// parallel region.
template <typename Function>
void parallelForChunks(size_t iterationCount, size_t workItemCount, Function function) {
    static_assert(std::is_nothrow_invocable_v<Function&, size_t, size_t>,
                  "parallelForChunks requires a noexcept function");
    if (iterationCount == 0) return;

    constexpr size_t minimumWorkItemsPerThread = 4096;
    constexpr int defaultMaximumThreadCount = 8;
    int threadCount = 1;
    if (!omp_in_parallel()) {
        const int runtimeMaximum = std::max(1, omp_get_max_threads());
        const char* configuredThreadCount = std::getenv("OMP_NUM_THREADS");
        const int maximum = configuredThreadCount != nullptr && configuredThreadCount[0] != '\0'
                                ? runtimeMaximum
                                : std::min(runtimeMaximum, defaultMaximumThreadCount);
        const size_t usefulThreadCount =
            std::max(size_t{1}, workItemCount / minimumWorkItemsPerThread +
                                    size_t{workItemCount % minimumWorkItemsPerThread != 0});
        threadCount = static_cast<int>(std::min(usefulThreadCount, static_cast<size_t>(maximum)));
    }

    if (threadCount > 1) {
#pragma omp parallel for schedule(static, 1) num_threads(threadCount)
        for (int chunkIndex = 0; chunkIndex < threadCount; ++chunkIndex) {
            const size_t chunk = static_cast<size_t>(chunkIndex);
            const size_t chunks = static_cast<size_t>(threadCount);
            const size_t baseSize = iterationCount / chunks;
            const size_t extraItems = iterationCount % chunks;
            const size_t begin = chunk * baseSize + std::min(chunk, extraItems);
            function(begin, begin + baseSize + size_t{chunk < extraItems});
        }
        return;
    }
    function(size_t{0}, iterationCount);
}

}  // namespace roc::hostnumerics::amd_gpu_layout::detail
