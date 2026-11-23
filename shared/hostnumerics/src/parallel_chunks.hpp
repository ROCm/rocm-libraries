// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace roc::hostnumerics::detail {

// Number of threads to use for `itemCount` independent items: as many as give
// each thread at least 4096 items, and at most 8 unless OMP_NUM_THREADS is set.
// Inside an OpenMP parallel region this is 1, so nested calls run serially.
inline size_t parallelChunkCount(size_t itemCount) {
    constexpr size_t minimumItemsPerThread = 4096;
    constexpr int defaultMaximumThreadCount = 8;
    if (omp_in_parallel()) return 1;
    const int runtimeMaximum = std::max(1, omp_get_max_threads());
    const char* configuredThreadCount = std::getenv("OMP_NUM_THREADS");
    const int maximum = configuredThreadCount != nullptr && configuredThreadCount[0] != '\0'
                            ? runtimeMaximum
                            : std::min(runtimeMaximum, defaultMaximumThreadCount);
    return std::clamp(itemCount / minimumItemsPerThread, size_t{1}, static_cast<size_t>(maximum));
}

// Calls function(chunk, begin, end) once for each chunk in [0, chunkCount).
// The ranges [begin, end) are disjoint, differ in length by at most one and
// cover [0, itemCount). Chunks run on separate OpenMP threads when chunkCount
// exceeds 1; `chunk` lets the caller give each one preallocated state.
// `function` must be noexcept, because an exception cannot leave an OpenMP
// parallel region.
template <typename Function>
void parallelForChunks(size_t itemCount, size_t chunkCount, Function function) {
    static_assert(std::is_nothrow_invocable_v<Function&, size_t, size_t, size_t>,
                  "parallelForChunks requires a noexcept function");
    assert(chunkCount > 0 && chunkCount <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const size_t baseSize = itemCount / chunkCount;
    const size_t extraItems = itemCount % chunkCount;
    const auto runChunk = [&](size_t chunk) noexcept {
        const size_t begin = chunk * baseSize + std::min(chunk, extraItems);
        function(chunk, begin, begin + baseSize + size_t{chunk < extraItems});
    };
    if (chunkCount == 1) {
        runChunk(0);
        return;
    }
    const int threadCount = static_cast<int>(chunkCount);
    // MSVC's OpenMP 2.0 requires a signed loop index.
#pragma omp parallel for schedule(static, 1) num_threads(threadCount)
    for (int chunk = 0; chunk < threadCount; ++chunk) runChunk(static_cast<size_t>(chunk));
}

}  // namespace roc::hostnumerics::detail
